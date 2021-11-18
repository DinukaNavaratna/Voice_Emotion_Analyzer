$(document).ready(function () {

    "use strict";

    // Preloader
    var preloader = $('.preloader');
    $(window).load(function () {
        preloader.remove();
    });

    // Navbar Collapse Button

    $(".button-collapse").sideNav();

    // Slider

    $('.slider').slider({
        full_width: true
    });

    // Full Screen Parallax

    var slideHeight = $(window).height();
    $('.full-screen').css('height', slideHeight);
    $(window).resize(function () {
        $('.full-screen').css('height', slideHeight);
    });

    // Navbar-Collapse (when scrolling)

    $(window).scroll(function () {
        if ($("header").offset().top > 50) {
            $("nav").addClass("top-nav-collapse");
        } else {
            $("nav").removeClass("top-nav-collapse");
        }
    });

    // Page Scroll

    $(function () {
        $('a.page-scroll').bind('click', function (event) {
            var $anchor = $(this);
            $('html, body').stop().animate({
                scrollTop: $($anchor.attr('href')).offset().top
            }, 1500, 'easeInOutExpo');
            event.preventDefault();
        });
    });

    // Owl Carousel (Testimonials)

    $("#owl-testimonials").owlCarousel({

        navigation: false, // hide next and prev buttons
        slideSpeed: 300,
        paginationSpeed: 400,
        singleItem: true

    });


    // Portfolio Filtre

    $('#portfolio-items').mixItUp({
        animation: {
            effects: 'fade rotateY(-25deg)'
        }
    });

    // Countdown

    $('#fun-facts').bind('inview', function (event, visible, visiblePartX, visiblePartY) {
        if (visible) {
            $(this).find('.timer').each(function () {
                var $this = $(this);
                $({
                    Counter: 0
                }).animate({
                    Counter: $this.text()
                }, {
                    duration: 4000,
                    easing: 'swing',
                    step: function () {
                        $this.text(Math.ceil(this.Counter));
                    }
                });
            });
            $(this).unbind('inview');
        }
    });

    // Collapsible 

    $('.collapsible').collapsible({
        accordion: false // A setting that changes the collapsible behavior to expandable instead of the default accordion style
    });

    // Google Map

    google.maps.event.addDomListener(window, 'load', init);

    function init() {
        // Basic options for a simple Google Map
        // For more options see: https://developers.google.com/maps/documentation/javascript/reference#MapOptions
        var mapOptions = {
            // How zoomed in you want the map to start at (always required)
            zoom: 15,

            // The latitude and longitude to center the map (always required)
            center: new google.maps.LatLng(33.5912284, -7.5210958, 17.18), // Casablanca

            // Disables the default Google Maps UI components
            disableDefaultUI: true,
            scrollwheel: false,

            // How you would like to style the map. 
            // This is where you would paste any style found on Snazzy Maps.
            styles: [{
                featureType: "administrative",
                elementType: "labels.text.fill",
                stylers: [{
                    color: "#444444"
                    }]
                }, {
                featureType: "administrative.country",
                elementType: "geometry.fill",
                stylers: [{
                    visibility: "on"
                    }]
                }, {
                featureType: "administrative.province",
                elementType: "labels.icon",
                stylers: [{
                    hue: "#ff0000"
                    }, {
                    visibility: "on"
                    }]
                }, {
                featureType: "landscape",
                elementType: "all",
                stylers: [{
                    color: "#f2f2f2"
                    }]
                }, {
                featureType: "poi",
                elementType: "all",
                stylers: [{
                    visibility: "off"
                    }]
                }, {
                featureType: "road",
                elementType: "all",
                stylers: [{
                    saturation: -100
                    }, {
                    lightness: 45
                    }]
                }, {
                featureType: "road.highway",
                elementType: "all",
                stylers: [{
                    visibility: "simplified"
                    }]
                }, {
                featureType: "road.arterial",
                elementType: "labels.icon",
                stylers: [{
                    visibility: "off"
                    }]
                }, {
                featureType: "transit",
                elementType: "all",
                stylers: [{
                    visibility: "off"
                    }]
                }, {
                featureType: "water",
                elementType: "all",
                stylers: [{
                    color: "#46bcec"
                    }, {
                    visibility: "on"
                    }]
                }]
        };

        // Get the HTML DOM element that will contain your map 
        // We are using a div with id="map" seen below in the <body>
        var mapElement = document.getElementById('map');

        // Create the Google Map using out element and options defined above
        var map = new google.maps.Map(mapElement, mapOptions);
        var myLatLng = new google.maps.LatLng(33.592501, -7.522318);
        // Custom Map Marker Icon - Customize the map-marker.png file to customize your icon
        var marker = new google.maps.Marker({
            position: myLatLng,
            map: map,
            title: 'Hello World!'
        });
    }

    particlesJS("particles-js", {
        "particles": {
            "number": {
                "value": 80,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": "#ffffff"
            },
            "shape": {
                "type": "circle",
                "stroke": {
                    "width": 0,
                    "color": "#000000"
                },
                "polygon": {
                    "nb_sides": 5
                },
                "image": {
                    "src": "img/github.svg",
                    "width": 100,
                    "height": 100
                }
            },
            "opacity": {
                "value": 0.5,
                "random": false,
                "anim": {
                    "enable": false,
                    "speed": 1,
                    "opacity_min": 0.1,
                    "sync": false
                }
            },
            "size": {
                "value": 3,
                "random": true,
                "anim": {
                    "enable": false,
                    "speed": 40,
                    "size_min": 0.1,
                    "sync": false
                }
            },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": "#ffffff",
                "opacity": 0.4,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": 6,
                "direction": "none",
                "random": false,
                "straight": false,
                "out_mode": "out",
                "bounce": false,
                "attract": {
                    "enable": false,
                    "rotateX": 600,
                    "rotateY": 1200
                }
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "bubble"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                },
                "resize": true
            },
            "modes": {
                "grab": {
                    "distance": 400,
                    "line_linked": {
                        "opacity": 1
                    }
                },
                "bubble": {
                    "distance": 400,
                    "size": 43.956043956043956,
                    "duration": 2,
                    "opacity": 1,
                    "speed": 3
                },
                "repulse": {
                    "distance": 200,
                    "duration": 0.4
                },
                "push": {
                    "particles_nb": 4
                },
                "remove": {
                    "particles_nb": 2
                }
            }
        },
        "retina_detect": true
    });
    var count_particles, stats, update;
    stats = new Stats;
    stats.setMode(0);
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.left = '0px';
    stats.domElement.style.top = '0px';
    document.body.appendChild(stats.domElement);
    count_particles = document.querySelector('.js-count-particles');
    update = function () {
        stats.begin();
        stats.end();
        if (window.pJSDom[0].pJS.particles && window.pJSDom[0].pJS.particles.array) {
            count_particles.innerText = window.pJSDom[0].pJS.particles.array.length;
        }
        requestAnimationFrame(update);
    };
    requestAnimationFrame(update);;

});
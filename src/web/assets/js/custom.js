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

});
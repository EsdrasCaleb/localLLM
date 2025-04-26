package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_spRatingBelow_13_2_Test {

    @Test
    void spRatingBelow_returnsCorrectRating() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.spRatingBelow("Rating");
        assertEquals("Rating", subscription.spRatingBelow());
    }
}

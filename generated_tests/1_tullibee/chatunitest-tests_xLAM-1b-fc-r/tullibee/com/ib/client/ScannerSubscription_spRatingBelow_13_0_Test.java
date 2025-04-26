package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_spRatingBelow_13_0_Test {

    @Test
    void spRatingBelow() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.spRatingBelow("Rating Below");
        assertEquals("Rating Below", scanner.spRatingBelow());
    }
}

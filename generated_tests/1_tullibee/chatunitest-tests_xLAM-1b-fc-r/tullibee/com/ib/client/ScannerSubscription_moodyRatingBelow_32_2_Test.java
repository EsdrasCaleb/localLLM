package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_moodyRatingBelow_32_2_Test {

    @Test
    void testMoodyRatingBelow() {
        ScannerSubscription subscription = new ScannerSubscription();
        String input = "test";
        subscription.moodyRatingBelow(input);
        assertEquals(input, subscription.moodyRatingBelow());
    }
}

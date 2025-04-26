package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_aboveVolume_6_2_Test {

    @Test
    void testAboveVolume() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expected = 100;
        subscription.aboveVolume(expected);
        assertEquals(expected, subscription.aboveVolume());
    }
}

package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_averageOptionVolumeAbove_7_4_Test {

    @Test
    void averageOptionVolumeAboveTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.averageOptionVolumeAbove(100);
        assertEquals(100, subscription.averageOptionVolumeAbove());
    }
}

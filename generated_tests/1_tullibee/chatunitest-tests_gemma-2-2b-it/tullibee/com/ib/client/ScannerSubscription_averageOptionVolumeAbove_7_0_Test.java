package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    @Test
    void testAverageOptionVolumeAbove() {
        ScannerSubscription scannerSubscription = mock(ScannerSubscription.class);
        when(scannerSubscription.averageOptionVolumeAbove()).thenReturn(10);
        int result = scannerSubscription.averageOptionVolumeAbove();
        assertEquals(10, result);
    }
}

package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_28_4_Test {

    @Test
    public void testAverageOptionVolumeAbove() {
        ScannerSubscription scannerSubscription = Mockito.spy(new ScannerSubscription());
        when(scannerSubscription.numberOfRows()).thenReturn(10);
        when(scannerSubscription.instrument()).thenReturn("Instrument");
        when(scannerSubscription.locationCode()).thenReturn("Location");
        when(scannerSubscription.scanCode()).thenReturn("Scan");
        when(scannerSubscription.abovePrice()).thenReturn(100.0);
        when(scannerSubscription.aboveVolume()).thenReturn(100);
        when(scannerSubscription.marketCapAbove()).thenReturn(100000.0);
        when(scannerSubscription.couponRateAbove()).thenReturn(0.05);
        scannerSubscription.averageOptionVolumeAbove(150);
        assertEquals(150, scannerSubscription.averageOptionVolumeAbove());
    }
}

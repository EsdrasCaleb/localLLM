package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_2_Test {

    @Test
    public void testAverageOptionVolumeAbove_NoOptionVolumeAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        int result = scannerSubscription.averageOptionVolumeAbove();
        assertEquals(Integer.MAX_VALUE, result);
    }

    @Test
    public void testAverageOptionVolumeAbove_OptionVolumeAbove() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(100.0);
        scannerSubscription.averageOptionVolumeAbove();
        scannerSubscription.aboveVolume(1000);
        scannerSubscription.averageOptionVolumeAbove();
        scannerSubscription.aboveVolume(500);
        scannerSubscription.averageOptionVolumeAbove();
        scannerSubscription.aboveVolume(2000);
        scannerSubscription.averageOptionVolumeAbove();
        scannerSubscription.aboveVolume(1000);
        scannerSubscription.averageOptionVolumeAbove();
        int result = scannerSubscription.averageOptionVolumeAbove();
        assertEquals(1000, result);
    }

    @Test
    public void testAverageOptionVolumeAbove_NullInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertThrows(NullPointerException.class, () -> scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_InvalidInput() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.abovePrice(Double.NaN);
        scannerSubscription.averageOptionVolumeAbove();
        scannerSubscription.abovePrice(-Double.MAX_VALUE);
        scannerSubscription.averageOptionVolumeAbove();
        scannerSubscription.abovePrice(Double.MAX_VALUE);
        scannerSubscription.averageOptionVolumeAbove();
        int result = scannerSubscription.averageOptionVolumeAbove();
        assertEquals(Integer.MAX_VALUE, result);
    }
}

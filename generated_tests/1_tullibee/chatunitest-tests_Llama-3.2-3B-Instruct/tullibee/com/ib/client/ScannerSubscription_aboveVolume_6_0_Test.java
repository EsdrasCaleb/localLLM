package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_6_0_Test {

    @Test
    public void testAboveVolume_defaultValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        assertEquals(Integer.MAX_VALUE, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolume_setValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.aboveVolume(100);
        assertEquals(100, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolume_setValue_maxValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.aboveVolume(Integer.MAX_VALUE);
        assertEquals(Integer.MAX_VALUE, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolume_setValue_negativeValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.aboveVolume(-100);
        assertEquals(-100, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolume_setValue_zeroValue() {
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        scannerSubscription.aboveVolume(0);
        assertEquals(0, scannerSubscription.aboveVolume());
    }
}

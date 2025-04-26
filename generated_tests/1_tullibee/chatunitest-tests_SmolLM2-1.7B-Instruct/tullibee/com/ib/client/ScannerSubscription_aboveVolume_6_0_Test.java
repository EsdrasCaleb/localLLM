package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_6_0_Test {

    @Test
    public void testAboveVolume() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        subscription.averageOptionVolumeAbove(5);
        assertEquals(5, subscription.aboveVolume());
        assertNotEquals(10, subscription.aboveVolume());
    }

    @Test
    public void testAboveVolumeZero() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(0);
        subscription.averageOptionVolumeAbove(5);
        assertEquals(0, subscription.aboveVolume());
        assertNotEquals(5, subscription.aboveVolume());
    }

    @Test
    public void testAboveVolumeNegative() {
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        subscription.averageOptionVolumeAbove(-5);
        assertEquals(-5, subscription.aboveVolume());
        assertNotEquals(10, subscription.aboveVolume());
    }
}

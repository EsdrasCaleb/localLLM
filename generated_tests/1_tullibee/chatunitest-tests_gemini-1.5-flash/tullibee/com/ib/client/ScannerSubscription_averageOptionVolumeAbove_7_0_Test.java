package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_7_0_Test {

    @Test
    void testAverageOptionVolumeAbove_defaultValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        assertEquals(Integer.MAX_VALUE, subscription.averageOptionVolumeAbove());
    }

    @Test
    void testAverageOptionVolumeAbove_setValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedVolume = 1000;
        subscription.averageOptionVolumeAbove(expectedVolume);
        assertEquals(expectedVolume, subscription.averageOptionVolumeAbove());
    }

    @Test
    void testAverageOptionVolumeAbove_setValueThenChange() {
        ScannerSubscription subscription = new ScannerSubscription();
        int initialVolume = 1000;
        int changedVolume = 2000;
        subscription.averageOptionVolumeAbove(initialVolume);
        subscription.averageOptionVolumeAbove(changedVolume);
        assertEquals(changedVolume, subscription.averageOptionVolumeAbove());
    }

    @Test
    void testAverageOptionVolumeAbove_setNegativeValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        int negativeVolume = -1000;
        subscription.averageOptionVolumeAbove(negativeVolume);
        assertEquals(negativeVolume, subscription.averageOptionVolumeAbove());
    }

    @Test
    void testAverageOptionVolumeAbove_setZeroValue() {
        ScannerSubscription subscription = new ScannerSubscription();
        int zeroVolume = 0;
        subscription.averageOptionVolumeAbove(zeroVolume);
        assertEquals(zeroVolume, subscription.averageOptionVolumeAbove());
    }
}

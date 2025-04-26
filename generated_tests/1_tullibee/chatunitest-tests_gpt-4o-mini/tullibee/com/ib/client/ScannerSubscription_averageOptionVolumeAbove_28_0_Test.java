package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_28_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAverageOptionVolumeAbove_SetPositiveValue() {
        int volume = 1000;
        scannerSubscription.averageOptionVolumeAbove(volume);
        assertEquals(volume, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_SetZeroValue() {
        int volume = 0;
        scannerSubscription.averageOptionVolumeAbove(volume);
        assertEquals(volume, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_SetNegativeValue() {
        int volume = -500;
        scannerSubscription.averageOptionVolumeAbove(volume);
        assertEquals(volume, scannerSubscription.averageOptionVolumeAbove());
    }

    @Test
    public void testAverageOptionVolumeAbove_SetMaxIntegerValue() {
        int volume = Integer.MAX_VALUE;
        scannerSubscription.averageOptionVolumeAbove(volume);
        assertEquals(volume, scannerSubscription.averageOptionVolumeAbove());
    }
}

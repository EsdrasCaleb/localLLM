package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_28_0_Test {

    @Test
    public void averageOptionVolumeAboveTest() {
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedVolume = 100;
        subscription.averageOptionVolumeAbove(expectedVolume);
        assertEquals(expectedVolume, subscription.averageOptionVolumeAbove());
    }
}

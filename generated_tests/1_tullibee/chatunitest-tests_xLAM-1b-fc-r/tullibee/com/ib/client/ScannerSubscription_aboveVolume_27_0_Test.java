package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_27_0_Test {

    @Test
    public void testAboveVolume() {
        ScannerSubscription subscription = new ScannerSubscription();
        int volume = 100;
        subscription.aboveVolume(volume);
        assertEquals(volume, subscription.aboveVolume());
    }
}

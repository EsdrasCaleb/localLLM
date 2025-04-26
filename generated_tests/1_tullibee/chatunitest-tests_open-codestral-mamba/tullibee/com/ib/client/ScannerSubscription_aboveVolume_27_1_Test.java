package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_27_1_Test {

    @Test
    public void testAboveVolume() {
        ScannerSubscription subscription = Mockito.spy(new ScannerSubscription());
        subscription.aboveVolume(100);
        Mockito.verify(subscription, Mockito.times(1)).aboveVolume(100);
        assertEquals(100, subscription.aboveVolume());
    }
}

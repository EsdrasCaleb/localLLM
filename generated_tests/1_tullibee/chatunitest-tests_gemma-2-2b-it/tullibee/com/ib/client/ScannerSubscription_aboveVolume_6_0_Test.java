package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_aboveVolume_6_0_Test {

    @Mock
    private ScannerSubscription scannerSubscription;

    @InjectMocks
    private ScannerSubscription scannerSubscriptionInstance;

    @Test
    public void aboveVolumeTest() {
        int expected = 100;
        scannerSubscriptionInstance.aboveVolume();
        assertEquals(expected, scannerSubscriptionInstance.aboveVolume());
    }
}

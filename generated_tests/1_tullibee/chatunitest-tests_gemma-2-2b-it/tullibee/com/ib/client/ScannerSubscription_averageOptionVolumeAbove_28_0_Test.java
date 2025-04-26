package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_averageOptionVolumeAbove_28_0_Test {

    ScannerSubscription scannerSubscription;

    @BeforeEach
    void setup() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    void averageOptionVolumeAbove() {
        int volume = 10;
        scannerSubscription.averageOptionVolumeAbove(volume);
        assertEquals(volume, scannerSubscription.averageOptionVolumeAbove());
    }
}

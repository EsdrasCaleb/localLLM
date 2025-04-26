package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_aboveVolume_6_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAboveVolume() {
        // Test default value
        assertEquals(Integer.MAX_VALUE, scannerSubscription.aboveVolume());
        // Test setting a new value
        int expectedVolume = 100;
        scannerSubscription.aboveVolume(expectedVolume);
        assertEquals(expectedVolume, scannerSubscription.aboveVolume());
    }
}

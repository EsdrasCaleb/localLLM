package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_aboveVolume_6_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testAboveVolumeDefault() {
        // Test the default value of m_aboveVolume
        assertEquals(Integer.MAX_VALUE, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolumeSetAndGet() {
        // Set a specific value for m_aboveVolume and test
        int testVolume = 1000;
        scannerSubscription.aboveVolume(testVolume);
        assertEquals(testVolume, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolumeNegativeValue() {
        // Set a negative value for m_aboveVolume and test
        int testVolume = -500;
        scannerSubscription.aboveVolume(testVolume);
        assertEquals(testVolume, scannerSubscription.aboveVolume());
    }

    @Test
    public void testAboveVolumeZero() {
        // Set the value of m_aboveVolume to zero and test
        int testVolume = 0;
        scannerSubscription.aboveVolume(testVolume);
        assertEquals(testVolume, scannerSubscription.aboveVolume());
    }
}

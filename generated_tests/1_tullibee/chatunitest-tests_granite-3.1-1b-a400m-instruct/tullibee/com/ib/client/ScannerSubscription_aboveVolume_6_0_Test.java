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
    public void testAboveVolume() {
        // Arrange
        scannerSubscription.instrument("ExampleStock");
        scannerSubscription.locationCode("NewYork");
        scannerSubscription.scannerSettingPairs("123");
        scannerSubscription.stockTypeFilter("Common");
        // Act
        int aboveVolume = scannerSubscription.aboveVolume();
        // Assert
        assertEquals(10, aboveVolume);
    }
}

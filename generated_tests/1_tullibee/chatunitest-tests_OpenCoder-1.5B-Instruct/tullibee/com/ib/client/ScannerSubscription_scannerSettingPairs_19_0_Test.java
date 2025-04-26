package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_19_0_Test {

    @Test
    public void testScannerSettingPairs() {
        // Arrange
        ScannerSubscription scannerSubscription = Mockito.mock(ScannerSubscription.class);
        when(scannerSubscription.scannerSettingPairs()).thenReturn("test");
        // Act
        String result = scannerSubscription.scannerSettingPairs();
        // Assert
        assertEquals("test", result);
    }
}

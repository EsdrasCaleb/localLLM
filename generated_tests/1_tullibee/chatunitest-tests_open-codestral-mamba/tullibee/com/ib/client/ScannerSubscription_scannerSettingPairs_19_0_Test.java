package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_scannerSettingPairs_19_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @Test
    public void testScannerSettingPairs() {
        // Arrange
        String expected = "ExpectedValue";
        scannerSubscription.scannerSettingPairs(expected);
        // Act
        String actual = scannerSubscription.scannerSettingPairs();
        // Assert
        assertEquals(expected, actual);
    }
}

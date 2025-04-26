package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scannerSettingPairs_19_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testScannerSettingPairs() {
        // Arrange
        String expectedScannerSettingPairs = "setting1=value1;setting2=value2";
        scannerSubscription.scannerSettingPairs(expectedScannerSettingPairs);
        // Act
        String actualScannerSettingPairs = scannerSubscription.scannerSettingPairs();
        // Assert
        assertEquals(expectedScannerSettingPairs, actualScannerSettingPairs);
    }
}

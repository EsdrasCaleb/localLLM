package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_scanCode_3_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testScanCode_WhenScanCodeIsNotSet_ShouldReturnNull() {
        // Act
        String result = scannerSubscription.scanCode();
        // Assert
        assertEquals(null, result);
    }

    @Test
    public void testScanCode_WhenScanCodeIsSet_ShouldReturnCorrectValue() {
        // Arrange
        String expectedScanCode = "ABC123";
        scannerSubscription.scanCode(expectedScanCode);
        // Act
        String result = scannerSubscription.scanCode();
        // Assert
        assertEquals(expectedScanCode, result);
    }
}

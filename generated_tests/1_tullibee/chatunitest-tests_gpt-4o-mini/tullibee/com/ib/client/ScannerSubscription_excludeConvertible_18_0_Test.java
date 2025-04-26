package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testExcludeConvertible_WhenNotSet_ShouldReturnNull() {
        // Arrange
        // No setup needed as m_excludeConvertible is initialized to null by default.
        // Act
        String result = scannerSubscription.excludeConvertible();
        // Assert
        assertEquals(null, result);
    }

    @Test
    public void testExcludeConvertible_WhenSetToValue_ShouldReturnValue() {
        // Arrange
        String expectedValue = "Exclude";
        scannerSubscription.excludeConvertible(expectedValue);
        // Act
        String result = scannerSubscription.excludeConvertible();
        // Assert
        assertEquals(expectedValue, result);
    }

    @Test
    public void testExcludeConvertible_WhenSetToEmptyString_ShouldReturnEmptyString() {
        // Arrange
        String expectedValue = "";
        scannerSubscription.excludeConvertible(expectedValue);
        // Act
        String result = scannerSubscription.excludeConvertible();
        // Assert
        assertEquals(expectedValue, result);
    }
}

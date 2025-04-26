package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testExcludeConvertible() throws Exception {
        // Arrange
        String expectedExcludeConvertible = "excludeConvertibleValue";
        scannerSubscription.excludeConvertible(expectedExcludeConvertible);
        // Act
        String result = scannerSubscription.excludeConvertible();
        // Assert
        assertEquals(expectedExcludeConvertible, result);
    }

    @Test
    public void testExcludeConvertible_DefaultValue() throws Exception {
        // Arrange
        // No value set, should return null
        // Act
        String result = scannerSubscription.excludeConvertible();
        // Assert
        assertEquals(null, result);
    }
}

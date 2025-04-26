package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testLocationCode() {
        // Arrange
        String expectedLocationCode = "US";
        scannerSubscription.locationCode(expectedLocationCode);
        // Act
        String actualLocationCode = scannerSubscription.locationCode();
        // Assert
        assertEquals(expectedLocationCode, actualLocationCode);
    }
}

package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_locationCode_2_2_Test {

    @Test
    public void testLocationCode() {
        // Arrange
        ScannerSubscription scannerSubscription = new ScannerSubscription();
        String testLocationCode = "TestLocationCode";
        // Act
        scannerSubscription.locationCode(testLocationCode);
        // Assert
        assertEquals(testLocationCode, scannerSubscription.locationCode());
    }
}

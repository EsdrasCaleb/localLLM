package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_numberOfRows_0_0_Test {

    @Test
    public void testNumberOfRows() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        subscription.numberOfRows(10);
        // Act
        int actualNumberOfRows = subscription.numberOfRows();
        // Assert
        assertEquals(10, actualNumberOfRows);
    }
}

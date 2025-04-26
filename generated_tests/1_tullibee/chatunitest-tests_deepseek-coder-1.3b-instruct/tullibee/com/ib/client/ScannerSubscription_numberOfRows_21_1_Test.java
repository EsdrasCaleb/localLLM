package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_21_1_Test {

    @Test
    public void testNumberOfRows() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        int expectedNumberOfRows = 10;
        // Act
        subscription.numberOfRows(expectedNumberOfRows);
        int actualNumberOfRows = subscription.numberOfRows();
        // Assert
        Assertions.assertEquals(expectedNumberOfRows, actualNumberOfRows);
    }
}

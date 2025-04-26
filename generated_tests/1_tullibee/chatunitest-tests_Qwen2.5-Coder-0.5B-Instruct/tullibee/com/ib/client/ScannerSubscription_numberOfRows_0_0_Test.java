package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ScannerSubscription_numberOfRows_0_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class ScannerSubscriptionNumberOfRowsTest {

        @Mock
        private ScannerSubscription scannerSubscription;

        @Test
        public void testNumberOfRows() {
            // Arrange
            Mockito.when(scannerSubscription.numberOfRows()).thenReturn(10);
            // Act
            int result = scannerSubscription.numberOfRows();
            // Assert
            assertEquals(10, result);
        }
    }
}

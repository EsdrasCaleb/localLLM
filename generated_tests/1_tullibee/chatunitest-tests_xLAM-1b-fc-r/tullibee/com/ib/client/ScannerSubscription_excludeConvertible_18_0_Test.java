package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    @Test
    public void testExcludeConvertible() {
        // Arrange
        ScannerSubscription subscription = new ScannerSubscription();
        // Act
        String result = subscription.excludeConvertible();
        // Assert
        assertEquals("test", result);
    }
}

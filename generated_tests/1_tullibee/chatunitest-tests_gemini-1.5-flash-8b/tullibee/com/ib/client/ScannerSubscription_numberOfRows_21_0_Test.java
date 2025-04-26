package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class ScannerSubscription_numberOfRows_21_0_Test {

    @Test
    void testNumberOfRows() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test case 1: Setting a valid positive number of rows
        subscription.numberOfRows(10);
        assertEquals(10, subscription.numberOfRows());
        // Test case 2: Setting a valid zero number of rows
        subscription.numberOfRows(0);
        assertEquals(0, subscription.numberOfRows());
        // Test case 3: Setting a valid negative number of rows
        subscription.numberOfRows(-5);
        assertEquals(-5, subscription.numberOfRows());
    }
}

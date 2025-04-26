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
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.numberOfRows(10);
        assertEquals(10, scanner.numberOfRows());
    }
}

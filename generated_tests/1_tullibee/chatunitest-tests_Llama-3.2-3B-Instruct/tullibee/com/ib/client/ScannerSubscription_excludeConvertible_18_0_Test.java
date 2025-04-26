package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    @Test
    public void testExcludeConvertible_NoInput() {
        ScannerSubscription scanner = new ScannerSubscription();
        String result = scanner.excludeConvertible();
        assertNull(result);
    }

    @Test
    public void testExcludeConvertible_ExcludeConvertible() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.excludeConvertible();
        String result = scanner.excludeConvertible();
        assertEquals("Exclude convertible securities from the scanner", result);
    }

    @Test
    public void testExcludeConvertible_IncludedConvertible() {
        ScannerSubscription scanner = new ScannerSubscription();
        scanner.excludeConvertible();
        scanner.excludeConvertible();
        String result = scanner.excludeConvertible();
        assertNull(result);
    }
}

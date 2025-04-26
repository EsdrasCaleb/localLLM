package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_1_Test {

    @Test
    void testExcludeConvertible() {
        ScannerSubscription sc = new ScannerSubscription();
        assertEquals("N/A", sc.excludeConvertible());
    }
}

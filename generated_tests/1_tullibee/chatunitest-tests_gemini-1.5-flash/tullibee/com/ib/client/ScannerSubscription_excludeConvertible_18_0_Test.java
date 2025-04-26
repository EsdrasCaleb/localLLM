package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_18_0_Test {

    @Test
    void testExcludeConvertible() {
        ScannerSubscription subscription = new ScannerSubscription();
        // Test case 1: Initially null
        assertNull(subscription.excludeConvertible(), "Initially, excludeConvertible should be null");
        // Test case 2: Set a value and retrieve it
        subscription.excludeConvertible("true");
        assertEquals("true", subscription.excludeConvertible(), "excludeConvertible should return the set value");
        // Test case 3: Set a different value and retrieve it
        subscription.excludeConvertible("false");
        assertEquals("false", subscription.excludeConvertible(), "excludeConvertible should return the updated value");
        // Test case 4: Set to null and retrieve it
        subscription.excludeConvertible(null);
        assertNull(subscription.excludeConvertible(), "excludeConvertible should return null after setting to null");
        // Test case 5: Set to empty string and retrieve it
        subscription.excludeConvertible("");
        assertEquals("", subscription.excludeConvertible(), "excludeConvertible should return empty string after setting to empty string");
    }
}

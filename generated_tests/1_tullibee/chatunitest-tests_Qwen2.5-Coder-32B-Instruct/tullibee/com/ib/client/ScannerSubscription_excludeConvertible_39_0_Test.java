package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_excludeConvertible_39_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testExcludeConvertible() throws Exception {
        // Given
        String excludeCriteria = "EXCLUDE_CRITERIA";
        String expectedExcludeCriteria = "EXCLUDE_CRITERIA";
        // When
        scannerSubscription.excludeConvertible(excludeCriteria);
        // Then
        // Using reflection to access the private field
        java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        field.setAccessible(true);
        String actualExcludeCriteria = (String) field.get(scannerSubscription);
        assertEquals(expectedExcludeCriteria, actualExcludeCriteria, "The excludeConvertible method should set the m_excludeConvertible field correctly.");
    }
}

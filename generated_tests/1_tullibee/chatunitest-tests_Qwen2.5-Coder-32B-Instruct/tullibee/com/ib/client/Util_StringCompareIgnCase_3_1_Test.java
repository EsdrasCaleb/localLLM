package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_StringCompareIgnCase_3_1_Test {

    @Test
    public void testStringCompareIgnCase() throws Exception {
        // Test case 1: Both strings are equal (ignoring case)
        assertEquals(0, Util.StringCompareIgnCase("Hello", "hello"));
        // Test case 2: lhs is lexicographically less than rhs (ignoring case)
        assertTrue(Util.StringCompareIgnCase("apple", "Banana") < 0);
        // Test case 3: lhs is lexicographically greater than rhs (ignoring case)
        assertTrue(Util.StringCompareIgnCase("Orange", "apple") > 0);
        // Test case 4: One string is null
        assertEquals(1, Util.StringCompareIgnCase("NonNull", null));
        assertEquals(-1, Util.StringCompareIgnCase(null, "NonNull"));
        assertEquals(0, Util.StringCompareIgnCase(null, null));
        // Test case 5: Strings with different cases and special characters
        assertEquals(0, Util.StringCompareIgnCase("Te$t123", "te$t123"));
        assertTrue(Util.StringCompareIgnCase("Te$t123", "Te$t124") < 0);
        assertTrue(Util.StringCompareIgnCase("Te$t125", "Te$t124") > 0);
        // Test case 6: NormalizeString method is mocked to return specific values
        try (MockedStatic<Util> utilMockedStatic = Mockito.mockStatic(Util.class)) {
            utilMockedStatic.when(() -> Util.NormalizeString("lhs")).thenReturn("mocked_lhs");
            utilMockedStatic.when(() -> Util.NormalizeString("rhs")).thenReturn("mocked_rhs");
            Method stringCompareIgnCaseMethod = Util.class.getDeclaredMethod("StringCompareIgnCase", String.class, String.class);
            stringCompareIgnCaseMethod.setAccessible(true);
            assertEquals("mocked_lhs".compareToIgnoreCase("mocked_rhs"), (int) stringCompareIgnCaseMethod.invoke(null, "lhs", "rhs"));
        }
    }
}

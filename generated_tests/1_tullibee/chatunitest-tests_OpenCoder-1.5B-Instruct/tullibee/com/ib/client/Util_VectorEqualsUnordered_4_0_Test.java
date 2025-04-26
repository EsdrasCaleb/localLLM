// Test method
package com.ib.client;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class Util_VectorEqualsUnordered_4_0_Test {

    @Mock
    private Vector<Object> vector;

    @Test
    public void testVectorEqualsUnordered() {
        List<Object> lhs = Arrays.asList(1, 2, 3);
        List<Object> rhs = Arrays.asList(1, 2, 3);
        assertTrue(vectorEqualsUnordered(lhs, rhs));
        lhs = Arrays.asList(1, 2, 3);
        rhs = Arrays.asList(1, 2, 4);
        assertFalse(vectorEqualsUnordered(lhs, rhs));
        lhs = Arrays.asList(1, 2, 3);
        rhs = Arrays.asList(3, 2, 1);
        assertFalse(vectorEqualsUnordered(lhs, rhs));
    }

    private boolean vectorEqualsUnordered(List<Object> lhs, List<Object> rhs) {
        // Buggy line
        // return lhs.containsAll(rhs) && rhs.containsAll(lhs);
        // Corrected line
        return lhs.containsAll(rhs) && rhs.containsAll(lhs);
    }
}

package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Util_VectorEqualsUnordered_4_3_Test {

    @Mock
    private Vector lhs;

    @Mock
    private Vector rhs;

    @BeforeEach
    public void setup() {
        // Fix: Use Vector instead of ArrayList
        lhs = new Vector();
        // Fix: Use Vector instead of ArrayList
        rhs = new Vector();
    }

    @Test
    public void testVectorEqualsUnordered_EmptyVectors_ReturnsTrue() {
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_SameElements_ReturnsTrue() {
        lhs.add(1);
        lhs.add(2);
        rhs.add(1);
        rhs.add(2);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_DifferentElements_ReturnsFalse() {
        lhs.add(1);
        lhs.add(2);
        rhs.add(2);
        rhs.add(3);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_NullVector_ReturnsFalse() {
        lhs = null;
        rhs.add(1);
        rhs.add(2);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    public void testVectorEqualsUnordered_DifferentSizes_ReturnsFalse() {
        lhs.add(1);
        lhs.add(2);
        rhs.add(1);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }
}

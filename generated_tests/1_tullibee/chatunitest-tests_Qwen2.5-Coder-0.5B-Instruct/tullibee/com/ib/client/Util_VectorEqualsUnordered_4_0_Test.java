package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    public void testVectorEqualsUnordered() {
        // Arrange
        Vector lhs = new Vector();
        Vector rhs = new Vector();
        // Act
        boolean result = Util.VectorEqualsUnordered(lhs, rhs);
        // Assert
        assertEquals(true, result);
    }
}

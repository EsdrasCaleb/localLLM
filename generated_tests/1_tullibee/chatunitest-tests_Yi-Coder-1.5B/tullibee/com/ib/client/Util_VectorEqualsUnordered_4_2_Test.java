package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Util_VectorEqualsUnordered_4_2_Test {

    // Test class
    @Test
    public void testVectorEqualsUnordered() {
        // given
        Vector lhs = new Vector();
        Vector rhs = new Vector();
        // when
        boolean result = Util.VectorEqualsUnordered(lhs, rhs);
        // then
        assert result;
    }
}

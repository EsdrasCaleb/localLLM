package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_VectorEqualsUnordered_4_1_Test {

    @Test
    void testVectorEqualsUnordered() {
        Util util = new Util();
        Vector lhs = new Vector();
        lhs.add(1);
        lhs.add(2);
        lhs.add(3);
        Vector rhs = new Vector();
        rhs.add(1);
        rhs.add(2);
        rhs.add(3);
        assertTrue(util.VectorEqualsUnordered(lhs, rhs));
    }
}

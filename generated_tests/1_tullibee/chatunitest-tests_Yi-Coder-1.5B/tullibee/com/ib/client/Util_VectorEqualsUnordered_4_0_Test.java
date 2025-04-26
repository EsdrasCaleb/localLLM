package com.ib.client;

import static org.junit.Assert.assertTrue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

@RunWith(MockitoJUnitRunner.class)
public class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    public void testVectorEqualsUnordered() {
        Vector lhs = new Vector();
        Vector rhs = new Vector();
        lhs.add("a");
        lhs.add("b");
        lhs.add("c");
        rhs.add("a");
        rhs.add("b");
        rhs.add("c");
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
        rhs.add("d");
        assertTrue(!Util.VectorEqualsUnordered(lhs, rhs));
        rhs.add("e");
        assertTrue(!Util.VectorEqualsUnordered(lhs, rhs));
        Vector rhs2 = new Vector();
        rhs2.add("a");
        rhs2.add("b");
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs2));
        rhs2.add("d");
        assertTrue(!Util.VectorEqualsUnordered(lhs, rhs2));
        rhs2.add("e");
        assertTrue(!Util.VectorEqualsUnordered(lhs, rhs2));
    }
}

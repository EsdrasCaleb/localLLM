package com.ib.client;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class Util_VectorEqualsUnordered_4_0_Test {

    @Test
    void vectorEqualsUnordered_sameElementsDifferentOrder_returnsTrue() {
        List<Integer> lhsList = new ArrayList<>();
        lhsList.add(1);
        lhsList.add(2);
        lhsList.add(3);
        List<Integer> rhsList = new ArrayList<>();
        rhsList.add(3);
        rhsList.add(2);
        rhsList.add(1);
        Vector lhs = new Vector<>(lhsList);
        Vector rhs = new Vector<>(rhsList);
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    void vectorEqualsUnordered_differentElements_returnsFalse() {
        List<Integer> lhsList = new ArrayList<>();
        lhsList.add(1);
        lhsList.add(2);
        List<Integer> rhsList = new ArrayList<>();
        rhsList.add(3);
        rhsList.add(4);
        Vector lhs = new Vector<>(lhsList);
        Vector rhs = new Vector<>(rhsList);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    void vectorEqualsUnordered_oneVectorNull_returnsFalse() {
        List<Integer> lhsList = new ArrayList<>();
        lhsList.add(1);
        lhsList.add(2);
        Vector lhs = new Vector<>(lhsList);
        Vector rhs = null;
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    void vectorEqualsUnordered_bothVectorsNull_returnsTrue() {
        Vector lhs = null;
        Vector rhs = null;
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    @Test
    void vectorEqualsUnordered_emptyVectors_returnsTrue() {
        Vector lhs = new Vector<>();
        Vector rhs = new Vector<>();
        assertTrue(Util.VectorEqualsUnordered(lhs, rhs));
    }

    // Example with different data types
    @Test
    void vectorEqualsUnordered_differentDataTypes_returnsFalse() {
        List<Integer> lhsList = new ArrayList<>();
        lhsList.add(1);
        List<String> rhsList = new ArrayList<>();
        rhsList.add("1");
        Vector lhs = new Vector<>(lhsList);
        Vector rhs = new Vector<>(rhsList);
        assertFalse(Util.VectorEqualsUnordered(lhs, rhs));
    }
}

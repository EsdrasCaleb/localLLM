package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringCompare_2_0_Test {

    @Test
    void StringCompare_Test() {
        Util util = new Util();
        String lhs = "abc";
        String rhs = "def";
        int result = util.StringCompare(lhs, rhs);
        assertEquals(result, -1);
    }
}

package com.ib.client;

import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class Util_StringIsEmpty_0_2_Test {

    @Test
    public void testStringIsEmpty() {
        // Arrange
        Util util = new Util();
        // Act
        String str = null;
        System.out.println(util.StringIsEmpty(str));
        // Assert
        // Expected: false
        assert (util.StringIsEmpty(str) == false);
        // Arrange
        String str2 = "Hello, World!";
        System.out.println(util.StringIsEmpty(str2));
        // Assert
        // Expected: true
        assert (util.StringIsEmpty(str2) == true);
        // Arrange
        String str3 = "";
        System.out.println(util.StringIsEmpty(str3));
        // Assert
        // Expected: true
        assert (util.StringIsEmpty(str3) == true);
        // Arrange
        String str4 = "abc";
        System.out.println(util.StringIsEmpty(str4));
        // Assert
        // Expected: false
        assert (util.StringIsEmpty(str4) == false);
        // Arrange
        String str5 = null;
        System.out.println(util.StringIsEmpty(str5));
        // Assert
        // Expected: false
        assert (util.StringIsEmpty(str5) == false);
        // Arrange
        String str6 = "abc";
        System.out.println(util.StringIsEmpty(str6));
        // Assert
        // Expected: false
        assert (util.StringIsEmpty(str6) == false);
        // Arrange
        String str7 = "abc";
        System.out.println(util.StringIsEmpty(str7));
        // Assert
        // Expected: true
        assert (util.StringIsEmpty(str7) == true);
    }
}

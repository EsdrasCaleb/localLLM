package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_main_7_0_Test {

    @Test
    public void testMain() {
        String[] args = new String[1];
        args[0] = "0123456789";
        Base64.main(args);
        // Add your assertions here to check if the output is as expected
    }
}

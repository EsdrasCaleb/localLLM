package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

// Test class
public class Base64_altBase64ToByteArray_0_3_Test {

    // "Hello World" in Base64
    private static final String TEST_STRING = "SGVsbG8gd29ybGQ=";

    // "Hello World" in byte array format
    private static final byte[] EXPECTED_BYTE_ARRAY = { 72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100 };
}

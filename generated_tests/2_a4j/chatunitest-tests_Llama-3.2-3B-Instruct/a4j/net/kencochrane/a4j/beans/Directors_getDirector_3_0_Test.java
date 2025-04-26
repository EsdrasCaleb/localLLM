package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class Directors_getDirector_3_0_Test {

    @Test
    public void testGetDirector_NullList_ThrowsNullPointerException() {
        Directors directors = null;
        assertThrows(NullPointerException.class, () -> directors.getDirector(0));
    }
}

package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class ModeList_addMode_1_1_Test {

    @Test
    void testAddMode() {
        ModeList modeList = new ModeList();
        Mode mode1 = new Mode();
        Mode mode2 = new Mode();
        modeList.addMode(mode1);
        modeList.addMode(mode2);
        assertEquals(2, modeList.modes.size());
        assertEquals(mode1, modeList.modes.get(0));
        assertEquals(mode2, modeList.modes.get(1));
    }
}

package com.hf.sfm.sfmis.personinfo.business;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.pdo.APersonInfo;
import com.hf.sfm.util.DaoFactory;

@ExtendWith(MockitoExtension.class)
public class PersonInfoMgr_deleteByIds_1_3_Test {

    @Test
    void deleteByIds_success() {
        PersonInfoMgr personInfoMgr = mock(PersonInfoMgr.class);
        String[] ids = { "1", "2", "3" };
        when(personInfoMgr.deleteByIds(ids)).thenReturn("1");
        assertEquals("1", personInfoMgr.deleteByIds(ids));
    }
}

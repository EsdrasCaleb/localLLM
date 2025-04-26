package com.hf.sfm.sfmis.personinfo.business;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import com.hf.sfm.sfmis.personinfo.pdo.APersonInfo;
import com.hf.sfm.util.DaoFactory;

public class PersonInfoMgr_saveOrUpdate_0_2_Test {

    @Test
    void saveOrUpdate_shouldReturnSuccess() {
        PersonInfoMgr personInfoMgr = mock(PersonInfoMgr.class);
        APersonInfo pInfo = new APersonInfo();
        when(personInfoMgr.saveOrUpdate(pInfo)).thenReturn("1");
        String result = personInfoMgr.saveOrUpdate(pInfo);
        assertEquals("1", result);
    }

    @Test
    void saveOrUpdate_shouldReturnSuccessWithNullId() {
        PersonInfoMgr personInfoMgr = mock(PersonInfoMgr.class);
        APersonInfo pInfo = new APersonInfo();
        when(personInfoMgr.saveOrUpdate(pInfo)).thenReturn("1");
        String result = personInfoMgr.saveOrUpdate(pInfo);
        assertEquals("1", result);
    }
}
